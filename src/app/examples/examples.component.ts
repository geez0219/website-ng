import { Component, OnInit, ViewEncapsulation, HostListener } from '@angular/core';
import { Location } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { ActivatedRoute } from '@angular/router';

import { Tutorial } from '../tutorial';
import { BehaviorSubject } from 'rxjs';

@Component({
  selector: 'app-examples',
  templateUrl: './examples.component.html',
  styleUrls: ['../tutorial/tutorial.component.css']
})
export class ApiExamples implements OnInit, OnDestroy {
  apiList: API[];
  selectedAPI: string;
  currentSelection: string;
  currentAPIText: string;
  
  routerSubscription: Subscription;

  treeControl: NestedTreeControl<API>;
  dataSource: MatTreeNestedDataSource<API>;

  screenWidth: number;
  private screenWidth$ = new BehaviorSubject<number>(window.innerWidth);
  
  @HostListener('window:resize', ['$event'])
  onResize(event) {
    this.screenWidth$.next(event.target.innerWidth);
  }

  constructor(private http: HttpClient,
              private router: Router,) { }

  ngOnInit() {
    this.treeControl = new NestedTreeControl<API>(node => node.children);
    this.dataSource = new MatTreeNestedDataSource<API>();

    this.getAPIStructure();

    this.routerSubscription = this.router.events.subscribe((e: any) => {
      if (e instanceof NavigationEnd) {
        this.parseURL();
      }
    });

    this.screenWidth$.subscribe(width => {
      this.screenWidth = width;
    });
  }

  ngOnDestroy() {
    if (this.routerSubscription) {
      this.routerSubscription.unsubscribe();
    }
  }

  hasChild = (_: number, node: API) => !!node.children && node.children.length > 0;

  flatten(arr) {
    var ret: API[] = [];
    for (let a of arr) {
      if (a.children) {
        ret = ret.concat(this.flatten(a.children));
      } else {
        ret = ret.concat(a);
      }
    }

    return ret;
  }

  getAPIStructure() {
    this.http.get('assets/examples/structure.json', {responseType: 'text'}).subscribe(data => {
      this.apiList = <API[]>JSON.parse(data);
      
      this.dataSource.data = this.apiList;
      this.treeControl.dataNodes = this.apiList;
      
      this.parseURL();
    });
  }

  private parseURL() {
    var pathComponents = this.router.url.split("/api/");
    var name = "";
    if (pathComponents.length > 1) {
      name = pathComponents[1].split("#")[0];
    }
    else {
      name = "";
    }

    if (name === "") {
      this.updateCurrentAPI(this.apiList[0].children[0]);

      this.treeControl.expand(this.treeControl.dataNodes[0]);
    }
    else {
      var a: API[] = this.flatten(this.apiList).filter(api => {
        var split: string[] = api.name.split("/");
        var matchName: string[] = (name + ".md").split("/");
        
        return split[split.length - 1] === matchName[matchName.length - 1];
      });

      this.updateAPIContent(a[0]);
      this.expandNodes(a[0], name);
    }
  }

  expandNodes(api: API, apiName: string) {
    var apiParts: Array<string> = apiName.split("/");
    apiParts.pop();

    if (apiParts.length === 1) {
      this.treeControl.expand(this.treeControl.dataNodes[0]);
    } else {
      var searchRange = this.treeControl.dataNodes;
      var searchName = apiParts[0];
      for (var i: number = 0; i < apiParts.length - 1; i++) {
        searchName = searchName + "." + apiParts[i + 1];
        var expandNode = searchRange.filter(api => api.displayName === searchName)[0];
        this.treeControl.expand(expandNode);

        searchRange = expandNode.children;
      }
    }
  }

  updateCurrentAPI(api: API) {
    var path = this.updateAPIContent(api);

    this.router.navigateByUrl('/api/' + path);
  }

  private updateAPIContent(api: API) {
    if (!api)
      this.router.navigate(['PageNotFound']);

    window.scroll(0, 0);

    this.selectedAPI = api.name;
    this.currentSelection = 'assets/api/' + api.name;
    var path = api.name.substring(0, api.name.length - 3);
    if (!path.startsWith("fe")) {
      path = "fe/" + path;
    }
    this.getSelectedAPIText();
    
    return path;
  }

  getSelectedAPIText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentAPIText = data;
    });
  }

}

