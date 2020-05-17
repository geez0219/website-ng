import { Component, OnInit, HostListener, ViewChild, ElementRef } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { ActivatedRoute, Router, UrlSegment } from '@angular/router';
import { BehaviorSubject } from 'rxjs';
import { Title } from '@angular/platform-browser';

import { NestedTreeControl } from '@angular/cdk/tree';
import { MatSidenav} from '@angular/material/sidenav';
import { MatTreeNestedDataSource } from '@angular/material/tree';

import { Example } from '../example';
import { GlobalService } from '../global.service';

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
  scrollCounter: number=0;
  scrollThreshold: number=20;
  segments: UrlSegment[];
  fragment: string;

  treeControl: NestedTreeControl<Example>;
  dataSource: MatTreeNestedDataSource<Example>;
  minWidth: number = 640;
  screenWidth: number;
  private screenWidth$ = new BehaviorSubject<number>(window.innerWidth);

  structureHeaderDict = {
    'Content-Type': 'application/json',
    'Accept': "application/json, text/plain",
    'Access-Control-Allow-Origin': '*'
  }
  structureRequestOptions = {
    headers: new HttpHeaders(this.structureHeaderDict),
  };

  contentHeaderDict = {
    'Accept': "application/json, text/plain",
    'Access-Control-Allow-Origin': '*'
  }
  contentRequestOptions = {
    responseType: 'text' as 'text',
    headers: new HttpHeaders(this.contentHeaderDict)
  };

  @ViewChild('sidenav', { static: true })
  sidenav: MatSidenav;

  @ViewChild('grippy', { static: true })
  grippy: ElementRef;

  @HostListener('window:resize', ['$event'])
  onResize(event) {
    this.screenWidth$.next(event.target.innerWidth);
  }

  @HostListener('window:scroll', ['$event'])
  onScroll(event) {
    if (this.scrollCounter < this.scrollThreshold) {
      this.scrollCounter += 1;
    }
  }

  constructor(private http: HttpClient,
              private router: Router,
              private route: ActivatedRoute,
              private title: Title,
              private globalService: GlobalService) { }

  ngOnInit() {
    this.treeControl = new NestedTreeControl<Example>(node => node.children);
    this.dataSource = new MatTreeNestedDataSource<Example>();

    this.route.url.subscribe((segments: UrlSegment[]) => {
      this.globalService.setLoading();

      this.segments = segments;
      this.getExampleStructure();
    });

    this.route.fragment.subscribe((fragment: string) => {
      this.fragment = fragment;
    })

    this.screenWidth$.subscribe(width => {
      this.screenWidth = width;
    });
  }

  ngAfterViewChecked() {
    /* Scroll to the fragment postition.
       The reason for adding a scroll threshold is to scroll to the fragment position after image loaded.
       Before the counter goes to scollThreshold, the scrolling postion will stick to fragment.
       This wook will trigger the onScroll event and trigger back to this funtion again.
       So the counter will goes extremely fast.
    */

    if (this.fragment && this.scrollCounter < this.scrollThreshold){
      if (document.querySelector('#' + this.fragment) != null){
        document.querySelector('#' + this.fragment).scrollIntoView();
        window.scrollBy(0, -90); // the offset of navbar height
      }
    }
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

  expandNodes(exampleName: string) {
    var exampleParts: Array<string> = exampleName.split("/");
    exampleParts.pop();

    var expandNode = this.exampleList.filter(example => example.name === exampleParts[0])[0];
    this.treeControl.expand(expandNode);
  }

  getExampleStructure() {
    if (this.exampleList) {
      this.loadSelectedExample();
    } else {
      this.http.get('assets/branches/r1.0/example/structure.json', this.structureRequestOptions).subscribe(data => {
        this.exampleList = <Example[]>(data);

        this.dataSource.data = this.exampleList;
        this.treeControl.dataNodes = this.exampleList;

        this.loadSelectedExample();
      },
      error => {
        console.error(error);
        this.globalService.resetLoading();
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
        this.globalService.resetLoading();
        this.router.navigate(['PageNotFound'], {replaceUrl:true});
      }
    }
  }

  private updateExampleContent(example: Example) {
    window.scroll(0, 0);

    this.selectedExample = example.name;
    this.currentSelection = 'assets/branches/r1.0/example/' + example.name;

    this.getSelectedExampleText();
    this.title.setTitle(example.displayName + " | Fastestimator");
  }

  getSelectedExampleText() {
    this.http.get(this.currentSelection, this.contentRequestOptions).subscribe(data => {
      this.currentExampleText = data;
      this.globalService.resetLoading();
    },
    error => {
      console.error(error);
      this.globalService.resetLoading();
      this.router.navigate(['PageNotFound'], {replaceUrl:true})
    });
  }

  getImageUrl() {
    if (this.sidenav.opened) {
      this.grippy.nativeElement.style.left = "19rem"
      return "url(../../assets/images/sidebar-grippy-hide.png)"
    } else {
      this.grippy.nativeElement.style.left = "0rem"
      return "url(../../assets/images/sidebar-grippy-show.png)"
    }
  }

  checkSidebar() {
    if (this.sidenav.opened) {
      this.grippy.nativeElement.style.backgroundImage = "url(../../assets/images/sidebar-grippy-hide.png)"
      this.grippy.nativeElement.style.left = "19rem"
    } else {
      this.grippy.nativeElement.style.backgroundImage = "url(../../assets/images/sidebar-grippy-show.png)"
      this.grippy.nativeElement.style.left = "0rem"
    }
  }

  toggleMenu() {
    this.sidenav.toggle();
    this.checkSidebar();
  }

  createRouterLink(url: string) {
    var components: Array<string> = url.substring(0, url.length - 3).split('/');
    var ret = ['/examples'];

    return ret.concat(components);
  }
}
