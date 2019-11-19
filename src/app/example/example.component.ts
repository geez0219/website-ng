import { Component, OnInit, HostListener } from '@angular/core';
import { HttpClient } from '@angular/common/http'; 
import { ActivatedRoute, Router, NavigationEnd, UrlSegment } from '@angular/router';
import { Subscription, BehaviorSubject } from 'rxjs';

import { NestedTreeControl } from '@angular/cdk/tree';
import { MatTreeNestedDataSource } from '@angular/material/tree';

import { Example } from '../example';

@Component({
  selector: 'app-example',
  templateUrl: './example.component.html',
  styleUrls: ['../api/api.component.css']
})
export class ExampleComponent implements OnInit {
  exampleList: Example[];
  selectedExample: string;
  currentSelection: string;
  currentExampleText: string;
  
  segments: UrlSegment[];

  treeControl: NestedTreeControl<Example>;
  dataSource: MatTreeNestedDataSource<Example>;

  screenWidth: number;
  private screenWidth$ = new BehaviorSubject<number>(window.innerWidth);
  
  @HostListener('window:resize', ['$event'])
  onResize(event) {
    this.screenWidth$.next(event.target.innerWidth);
  }

  constructor(private http: HttpClient,
              private router: Router,
              private route: ActivatedRoute) { }

  ngOnInit() {
    this.treeControl = new NestedTreeControl<Example>(node => node.children);
    this.dataSource = new MatTreeNestedDataSource<Example>();

    this.route.url.subscribe((segments: UrlSegment[]) => {
      this.segments = segments;
      this.getExampleStructure();
    });

    this.screenWidth$.subscribe(width => {
      this.screenWidth = width;
    });
  }

  hasChild = (_: number, node: Example) => !!node.children && node.children.length > 0;

  flatten(arr) {
    var ret: Example[] = [];
    for (let a of arr) {
      if (a.children) {
        ret = ret.concat(this.flatten(a.children));
      } else {
        ret = ret.concat(a);
      }
    }

    return ret;
  }

  getExampleStructure() {
    if (this.exampleList) {
      this.loadSelectedExample();
    } else {
      this.http.get('assets/example/structure.json', {responseType: 'text'}).subscribe(data => {
        this.exampleList = <Example[]>JSON.parse(data);
        
        this.dataSource.data = this.exampleList;
        this.treeControl.dataNodes = this.exampleList;
        
        this.loadSelectedExample();
      });
    }
  }

  private loadSelectedExample() {
    if (this.segments.length == 0) {
      this.updateExampleContent(this.exampleList[0].children[0]);
      this.treeControl.expand(this.treeControl.dataNodes[0]);
    }
    else {
      var e: Example[] = this.flatten(this.exampleList)
        .filter(example => 
          (this.segments.map(segment => segment.toString()).join('/') + ".md") === example.name);
      
      if (e.length > 0) {
        this.updateExampleContent(e[0]);
        this.expandNodes(e[0].name);
      } else {
        this.router.navigate(['PageNotFound']);
      }
    }
  }

  expandNodes(exampleName: string) {
    var exampleParts: Array<string> = exampleName.split("/");
    exampleParts.pop();
    
    var expandNode = this.exampleList.filter(example => example.name === exampleParts[0])[0];
    this.treeControl.expand(expandNode);
  }

  private updateExampleContent(example: Example) {
    window.scroll(0, 0);

    this.selectedExample = example.name;
    this.currentSelection = 'assets/example/' + example.name;
    
    this.getSelectedExampleText();
  }

  getSelectedExampleText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentExampleText = data;
    });
  }

  createRouterLink(url: string) {
    var components: Array<string> = url.substring(0, url.length - 3).split('/');
    var ret = ['/examples'];
    
    return ret.concat(components);;
  }
}